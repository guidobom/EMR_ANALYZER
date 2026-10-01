"""Clinical history facade: delegates extraction to the FHIR event registry."""

import difflib
import itertools
import json
import threading
from pathlib import Path
from typing import Callable, Optional

from ..config import (
    active_workspace,
    GOLDEN_FEWSHOT_ENABLED,
    GOLDEN_FEWSHOT_MAX_EXAMPLES,
)
from ..models.clinical_timeline import ClinicalTimelineEntry
from ..prompt_catalog import load_prompt


class ClinicalHistoryBuilder:
    """Build a strictly temporal clinical history from normalized texts."""

    def __init__(
        self,
        timeline_repo,
        document_repo,
        cs_repo,
        clinical_state_llm_client,
        audit_repo=None,
        registry_builder=None,
    ):
        self._timeline_repo = timeline_repo
        self._doc_repo = document_repo
        self._cs_repo = cs_repo
        self._llm = clinical_state_llm_client
        self._audit = audit_repo
        self._registry_builder = registry_builder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_atomic_evidence(
        self,
        patient_id: str,
        *,
        incremental: bool = True,
        num_workers: int = 1,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> dict:
        """Run phase 1 without creating or modifying clinical events."""

        if self._registry_builder is None:
            raise RuntimeError(
                "La pipeline evidence-based non è disponibile."
            )
        return self._registry_builder.extract_atomic_evidence(
            patient_id,
            incremental=incremental,
            num_workers=num_workers,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )

    def build_structured_events(
        self,
        patient_id: str,
        *,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> dict:
        """Run phase 2 from stored atomic evidence only."""

        if self._registry_builder is None:
            raise RuntimeError(
                "La pipeline evidence-based non è disponibile."
            )
        return self._registry_builder.build_structured_events(
            patient_id,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )

    def prepare_validation(self, patient_id: str) -> dict:
        """Run phase 3: prepare the independent human-review queue."""

        if self._registry_builder is None:
            raise RuntimeError(
                "La pipeline evidence-based non è disponibile."
            )
        return self._registry_builder.prepare_validation(patient_id)

    def registry_pipeline_status(self, patient_id: str) -> dict:
        if self._registry_builder is None or not patient_id:
            return {}
        return self._registry_builder.pipeline_status(patient_id)

    def build_from_documents(
        self,
        patient_id: str,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        generate_narrative: bool = False,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> dict:
        """Run the full pipeline and persist the result.

        If *generate_narrative* is True, also generate the narrative
        clinical profile via LLM and store it in ClinicalState.

        Returns a summary dict with keys *total_entries*, *deduplicated*,
        and *final_entries*.
        """
        if self._registry_builder is not None:
            result = self._registry_builder.build(
                patient_id, incremental=False, num_workers=1,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )
            if generate_narrative:
                self.generate_narrative(patient_id)
            return result
        raise RuntimeError("Il servizio di estrazione ICD-11 non è disponibile.")

    def build_incremental(
        self,
        patient_id: str,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        generate_narrative: bool = False,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> dict:
        """Process only NEW documents and merge with the existing registry.

        Documents whose IDs already appear in the timeline are skipped.
        New entries are appended, then the FULL registry is deduplicated.
        The save is atomic: old entries are only removed after the new
        batch has been successfully saved.
        """
        if self._registry_builder is not None:
            atomic_llm = getattr(
                self._registry_builder, "atomic_llm", None
            )
            workers = max(1, int(getattr(
                atomic_llm, "parallel_workers", 1
            ) or 1))
            result = self._registry_builder.build(
                patient_id, incremental=True, num_workers=workers,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )
            if generate_narrative:
                self.generate_narrative(patient_id)
            return result
        raise RuntimeError("Il servizio di estrazione ICD-11 non è disponibile.")

    def build_from_documents_parallel(
        self,
        patient_id: str,
        num_workers: int = 4,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        generate_narrative: bool = False,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> dict:
        """Parallel version — extracts from every document concurrently.

        Each document is processed **independently** (empty registry context)
        so the LLM calls can run in parallel.  A global deduplication pass
        afterwards merges observations that describe the same clinical fact.

        Returns the same summary dict as :meth:`build_from_documents`.
        """
        if self._registry_builder is not None:
            result = self._registry_builder.build(
                patient_id, incremental=False, num_workers=num_workers,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )
            if generate_narrative:
                self.generate_narrative(patient_id)
            return result
        raise RuntimeError("Il servizio di estrazione ICD-11 non è disponibile.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------



    def generate_narrative(self, patient_id: str) -> str:
        """Generate a narrative clinical profile from the stored timeline.

        Can be called independently after :meth:`build_from_documents`.
        Returns the narrative text and stores it in ``ClinicalState``.
        """
        entries = self._timeline_repo.get_by_patient(patient_id)
        if not entries:
            return ""

        entries_data = [
            {
                "date": e.date_observed,
                "category": e.category,
                "description": e.description,
                "status": e.status,
            }
            for e in sorted(entries, key=lambda e: e.date_observed)
        ]

        if not self._llm or not self._llm.is_available:
            # Fallback: simple concatenation
            lines = ["## Profilo Clinico Cronologico\n"]
            for e in sorted(entries, key=lambda x: x.date_observed):
                resolved = ""
                if e.date_resolved:
                    resolved = f" → risolto {e.date_resolved}"
                lines.append(
                    f"- **{e.date_observed}** [{e.category}] "
                    f"{e.description}{resolved}"
                )
            narrative = "\n".join(lines)
        else:
            system_prompt = load_prompt("narrative_profile_system")
            user_prompt = (
                load_prompt("narrative_profile_task")
                + "\n\nOSSERVAZIONI CLINICHE:\n"
                + json.dumps(entries_data, ensure_ascii=False, indent=2)
                + "\n"
            )
            try:
                narrative = self._llm.generate_text(user_prompt, system_prompt)
            except Exception:
                lines = ["## Profilo Clinico Cronologico\n"]
                for e in sorted(entries, key=lambda x: x.date_observed):
                    resolved = ""
                    if e.date_resolved:
                        resolved = f" → risolto {e.date_resolved}"
                    lines.append(
                        f"- **{e.date_observed}** [{e.category}] "
                        f"{e.description}{resolved}"
                    )
                narrative = "\n".join(lines)

        self._update_clinical_profile(patient_id, narrative)
        return narrative

    def _update_clinical_profile(
        self, patient_id: str, narrative: str
    ) -> None:
        """Store the narrative in ClinicalState.clinical_profile."""
        state = self._cs_repo.load(patient_id)
        if state is None:
            from ..models.clinical_state import ClinicalState
            state = ClinicalState(patient_id=patient_id)
        state.clinical_profile = narrative
        self._cs_repo.save(state)

    def clear_timeline(self, patient_id: str) -> None:
        """Delete all timeline entries for a patient."""
        self._timeline_repo.delete_by_patient(patient_id)

    def deduplicate_existing(self, patient_id: str) -> int:
        """Deduplicate the timeline entries already stored in the DB.

        Reads all entries, runs semantic dedup via LLM, and overwrites
        the registry with the deduplicated result.  Returns the number
        of duplicates removed.
        """
        if self._registry_builder is not None:
            before = self._timeline_repo.count_by_patient(patient_id)
            result = self._registry_builder.build_structured_events(
                patient_id
            )
            return max(0, before - int(result.get("final_entries", before)))

        entries = self._timeline_repo.get_by_patient(patient_id)
        if len(entries) <= 1:
            return 0

        final = self._deduplicate_all(entries)
        self._timeline_repo.replace_all_for_patient(patient_id, final)
        return len(entries) - len(final)

    def clear_narrative(self, patient_id: str) -> None:
        """Delete the clinical profile narrative, keeping the timeline."""
        state = self._cs_repo.load(patient_id)
        if state is not None:
            state.clinical_profile = ""
            self._cs_repo.save(state)

    # Conservative thresholds: only near-verbatim entries (same date +
    # category, ≥0.75 similarity) are merged deterministically.  The LLM
    # stage handles the semantic duplicates the string matcher misses.
    _DETERMINISTIC_DEDUP_THRESHOLD = 0.75
    # Laboratory twins (parser entry vs LLM extraction): same parameter +
    # date with loosely similar value text.
    _DEDUP_LAB_TWIN_SIMILARITY = 0.6
    # Close-date merges: same category, dates within ±7 days and
    # essentially identical wording — very conservative.
    _DEDUP_CLOSE_DATE_SIMILARITY = 0.95
    _DEDUP_CLOSE_DATE_DAYS = 7

    @staticmethod
    def _deterministic_dedup(
        entries: list[ClinicalTimelineEntry],
    ) -> list[ClinicalTimelineEntry]:
        """Deterministic dedup pass that runs before the LLM stage.

        Four conservative stages reduce the entry count (and thus the
        number of LLM dedup calls) without semantic risk:
        1. exact normalized matches (same date + category);
        2. near-verbatim matches (same date + category, SequenceMatcher
           ≥ threshold on normalized text);
        3. laboratory twins: same parameter + date, loosely similar value
           text (deterministic parser entry vs LLM extraction from the
           document text);
        4. close dates: same category, |Δdate| ≤ 7 days, similarity ≥ 0.95.

        The survivor is the longest / highest-confidence entry of the
        group; it accumulates the ``merged_into_ids`` (excluding its own
        id) and the sources of the fused entries.
        """
        if len(entries) <= 1:
            return entries

        threshold = ClinicalHistoryBuilder._DETERMINISTIC_DEDUP_THRESHOLD
        normalized = [
            ClinicalHistoryBuilder._normalize_description(e.description)
            for e in entries
        ]
        used: set[int] = set()

        def fold(group: list[int]) -> None:
            """Merge *group* into its survivor (mutates it in place)."""
            best_idx = max(
                group,
                key=lambda idx: (
                    len(entries[idx].description),
                    entries[idx].confidence,
                ),
            )
            survivor = entries[best_idx]
            merged_ids = [
                entries[idx].entry_id
                for idx in group if idx != best_idx
            ]
            all_doc_ids = list(survivor.source_document_ids)
            all_texts = list(survivor.source_texts)
            for idx in group:
                if idx == best_idx:
                    continue
                me = entries[idx]
                for did in me.source_document_ids:
                    if did not in all_doc_ids:
                        all_doc_ids.append(did)
                for txt in me.source_texts:
                    if txt not in all_texts:
                        all_texts.append(txt)
            survivor.source_document_ids = all_doc_ids
            survivor.source_texts = all_texts[:5]  # Cap at 5
            survivor.confidence = max(
                survivor.confidence,
                max(entries[idx].confidence for idx in group),
            )
            survivor.merged_into_ids = list(dict.fromkeys(
                survivor.merged_into_ids + merged_ids
            ))
            # Mark only the merged-away entries: the survivor stays eligible
            # so a later stage can merge it transitively into another group.
            for idx in group:
                if idx != best_idx:
                    used.add(idx)

        # --- stage 1: exact normalized matches (same date + category) ----
        exact_groups: dict[tuple, list[int]] = {}
        for i, entry in enumerate(entries):
            exact_groups.setdefault(
                (entry.date_observed, entry.category, normalized[i]), []
            ).append(i)
        for group in exact_groups.values():
            if len(group) > 1:
                fold(group)

        # --- stage 2: near-verbatim (same date + category) ---------------
        by_key: dict[tuple, list[int]] = {}
        for i, entry in enumerate(entries):
            by_key.setdefault(
                (entry.date_observed, entry.category), []
            ).append(i)
        for idxs in by_key.values():
            available = [idx for idx in idxs if idx not in used]
            for a, ia in enumerate(available):
                if ia in used:
                    continue
                group = [ia]
                for ib in available[a + 1:]:
                    if ib in used:
                        continue
                    ratio = difflib.SequenceMatcher(
                        None, normalized[ia], normalized[ib]
                    ).ratio()
                    if ratio >= threshold:
                        group.append(ib)
                if len(group) > 1:
                    fold(group)

        # --- stage 3: laboratory twins (same parameter + date) -----------
        for (date_obs, category), idxs in by_key.items():
            if category != "laboratory":
                continue
            available = [idx for idx in idxs if idx not in used]
            by_param: dict[str, list[int]] = {}
            for idx in available:
                param = ClinicalHistoryBuilder._lab_parameter(
                    entries[idx].description
                )
                by_param.setdefault(param, []).append(idx)
            for group_idxs in by_param.values():
                for a, ia in enumerate(group_idxs):
                    if ia in used:
                        continue
                    group = [ia]
                    value_a = ClinicalHistoryBuilder._lab_numeric_value(
                        entries[ia].description
                    )
                    for ib in group_idxs[a + 1:]:
                        if ib in used:
                            continue
                        value_b = ClinicalHistoryBuilder._lab_numeric_value(
                            entries[ib].description
                        )
                        if value_a is not None and value_a == value_b:
                            twin = True  # same parameter + same value
                        else:
                            ratio = difflib.SequenceMatcher(
                                None, normalized[ia], normalized[ib]
                            ).ratio()
                            twin = ratio >= (
                                ClinicalHistoryBuilder
                                ._DEDUP_LAB_TWIN_SIMILARITY
                            )
                        if twin:
                            group.append(ib)
                    if len(group) > 1:
                        fold(group)

        # --- stage 4: close dates (same category, ±7 days, ≥0.95) --------
        remaining = [idx for idx in range(len(entries)) if idx not in used]
        remaining.sort(key=lambda idx: (
            entries[idx].category, entries[idx].date_observed,
        ))
        for a, ia in enumerate(remaining):
            if ia in used:
                continue
            group = [ia]
            for ib in remaining[a + 1:]:
                if ib in used:
                    continue
                if entries[ib].category != entries[ia].category:
                    break  # sorted by category — no more candidates
                days = ClinicalHistoryBuilder._date_diff_days(
                    entries[ia].date_observed, entries[ib].date_observed
                )
                if days is None:
                    continue
                if days > ClinicalHistoryBuilder._DEDUP_CLOSE_DATE_DAYS:
                    break  # sorted by date — later ones are even farther
                ratio = difflib.SequenceMatcher(
                    None, normalized[ia], normalized[ib]
                ).ratio()
                if ratio >= (
                    ClinicalHistoryBuilder._DEDUP_CLOSE_DATE_SIMILARITY
                ):
                    group.append(ib)
            if len(group) > 1:
                fold(group)

        # Same stable output order as before: (date, category, description);
        # survivors have already been mutated in place.
        ordered = sorted(
            range(len(entries)),
            key=lambda idx: (
                entries[idx].date_observed,
                entries[idx].category,
                entries[idx].description,
            ),
        )
        return [entries[idx] for idx in ordered if idx not in used]

    @staticmethod
    def _normalize_description(text: str) -> str:
        """Case-fold and strip punctuation/whitespace for comparisons."""
        import re

        lowered = str(text or "").lower()
        lowered = re.sub(r"\s+", " ", lowered)
        lowered = re.sub(r"[^\w\s%./,:+-]", "", lowered)
        return lowered.strip(" .,:")

    @staticmethod
    def _lab_parameter(description: str) -> str:
        """Parameter-name prefix of a laboratory entry description.

        Parser entries are ``parametro: valore ...``; LLM entries are free
        form (``Emoglobina 10.2 g/dL ...``).  For the free form the second
        word is included only when it is part of the name (non-numeric),
        so ``Emoglobina 10.2`` resolves to ``emoglobina`` while
        ``Velocità eritrosedimentazione 45`` keeps both words.
        """
        text = str(description or "").strip()
        if ":" in text[:40]:
            return text.split(":", 1)[0].strip().lower()
        words = text.split()
        if not words:
            return ""
        first = words[0].lower()
        if len(words) > 1 and not (
            words[1][0].isdigit() or words[1][0] in "<>="
        ):
            return f"{first} {words[1].lower()}"
        return first

    @staticmethod
    def _lab_numeric_value(description: str) -> str | None:
        """First numeric token (operator + number) of a lab value."""
        import re

        match = re.search(
            r"([<>≤≥]?\s*\d+(?:[.,]\d+)?)", str(description or "")
        )
        if match is None:
            return None
        return match.group(1).replace(" ", "")

    @staticmethod
    def _date_diff_days(date_a: str, date_b: str) -> int | None:
        """Absolute day distance between two ISO dates; None if unparseable."""
        from datetime import datetime

        def parse(value: str):
            try:
                return datetime.strptime(str(value or "")[:10], "%Y-%m-%d")
            except ValueError:
                return None

        first, second = parse(date_a), parse(date_b)
        if first is None or second is None:
            return None
        return abs((second - first).days)

    def _apply_dedup_groups(
        self,
        entries: list[ClinicalTimelineEntry],
        dedup_result: dict,
    ) -> list[ClinicalTimelineEntry]:
        """Apply a dedup result to a list of timeline entries.

        Handles both the new ``groups`` contract (canonical synthesis with
        ``merged_into_ids`` provenance) and the legacy
        ``removed_entry_ids``/``enrichments`` contract for backward
        compatibility with older stored outputs.

        Returns the final list: merged entries are dropped, each survivor
        keeps the canonical description and accumulates the sources and the
        ``merged_into_ids`` provenance of the entries fused into it.
        """
        groups = dedup_result.get("groups") or []
        if not groups:
            # Legacy contract: drop removed ids, append enrichments.
            removed_ids = set(dedup_result.get("removed_entry_ids", []))
            enrichments = dedup_result.get("enrichments", {})
            final = []
            for entry in entries:
                if entry.entry_id in removed_ids:
                    continue
                enrichment = enrichments.get(entry.entry_id)
                if enrichment and isinstance(enrichment, str):
                    entry.description = (
                        f"{entry.description}\n\n"
                        f"[Integrazione: {enrichment}]"
                    )
                final.append(entry)
            return final

        by_id = {e.entry_id: e for e in entries}
        removed_ids: set[str] = set()
        known_ids = set(by_id)
        for g in groups:
            kept = g.get("kept_id")
            # A hallucinated kept_id (not among the entries) must not drop the
            # merged entries: without a survivor the clinical content would be
            # lost. Skip such groups entirely.
            if kept not in known_ids:
                continue
            removed_ids.update(
                m for m in g.get("merged_into_ids", []) if m != kept
            )

        final: list[ClinicalTimelineEntry] = []
        for entry in entries:
            if entry.entry_id in removed_ids:
                continue
            group = next(
                (g for g in groups if g.get("kept_id") == entry.entry_id),
                None,
            )
            if group:
                canonical = group.get("canonical_description")
                if canonical and isinstance(canonical, str):
                    entry.description = canonical

                merged = [
                    m for m in group.get("merged_into_ids", [])
                    if m in by_id
                ]
                entry.merged_into_ids = list(dict.fromkeys(
                    entry.merged_into_ids + merged
                ))
                for mid in merged:
                    me = by_id[mid]
                    for did in me.source_document_ids:
                        if did not in entry.source_document_ids:
                            entry.source_document_ids.append(did)
                    for txt in me.source_texts:
                        if txt not in entry.source_texts:
                            entry.source_texts.append(txt)

                # Date: prefer the group's (LLM-selected, earliest/most
                # precise), then the survivor's own, then the earliest
                # among the merged entries.
                group_date = group.get("date_observed")
                if group_date:
                    # Normalize like extraction: the LLM may return free-form
                    # dates ("febbraio 2024") that would break chronological
                    # ordering. Fall back to the survivor's own date.
                    entry.date_observed = self._normalize_date_observed(
                        group_date, entry.date_observed
                    )
                elif not entry.date_observed:
                    merged_dates = [
                        by_id[m].date_observed
                        for m in merged if by_id[m].date_observed
                    ]
                    if merged_dates:
                        entry.date_observed = min(merged_dates)

            final.append(entry)
        return final

    def _deduplicate_all(
        self, entries: list[ClinicalTimelineEntry],
    ) -> list[ClinicalTimelineEntry]:
        """Full dedup pipeline: deterministic pre-filter + LLM groups.

        The deterministic pre-filter collapses near-verbatim entries first
        (conservative, cheap), then the LLM groups the remaining semantic
        duplicates and synthesises a canonical description per group.
        """
        deduped = self._deterministic_dedup(entries)
        entries_dicts = [e.to_dict() for e in deduped]
        dedup_result = self._llm.deduplicate_timeline(entries_dicts)
        return self._apply_dedup_groups(deduped, dedup_result)

    @staticmethod
    def _normalize_date_observed(value, document_date: str = "") -> str:
        """Normalize a ``date_observed`` extracted by the LLM.

        Accepts ISO ``YYYY-MM-DD`` / ``YYYY-MM`` (month precision) and
        European ``DD/MM/YYYY`` formats.  When the value is empty or not
        parseable — e.g. the LLM returned ``None`` or a free-form phrase —
        it falls back to the document date, so no registry entry is ever
        left without a temporal anchor.
        """
        import re
        from datetime import datetime

        if isinstance(value, str):
            value = value.strip()
        if not value:
            return document_date or ""

        # ISO YYYY-MM-DD or YYYY-MM (already normalized)
        m = re.match(r"^(\d{4})-(\d{1,2})(?:-(\d{1,2}))?$", value)
        if m:
            year, month, day = m.groups()
            try:
                if day:
                    datetime(int(year), int(month), int(day))
                    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
                datetime(int(year), int(month), 1)
                return f"{int(year):04d}-{int(month):02d}"
            except ValueError:
                pass

        # European DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
        m = re.match(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})$", value)
        if m:
            day, month, year = m.groups()
            try:
                datetime(int(year), int(month), int(day))
                return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
            except ValueError:
                pass

        return document_date or ""


    @staticmethod
    def _get_normalized_text_path(patient_id: str, doc_id: str) -> Path:
        return active_workspace.path / patient_id / "extraction" / f"{doc_id}.md"

    @staticmethod
    def _format_registry_context(entries: list[dict]) -> str:
        """Format the running registry as a compact context string.

        Only the last 50 entries are included to stay within the LLM
        context window.
        """
        if not entries:
            return ""

        recent = entries[-50:]
        lines = []
        for e in recent:
            date = e.get("date_observed", "?")
            resolved = ""
            if e.get("date_resolved"):
                resolved = f" → {e['date_resolved']}"
            lines.append(
                f"[{date}{resolved}] [{e.get('category', '?')}] "
                f"{e.get('description', '')}"
            )
        return "\n".join(lines)

    def _load_golden_examples(self, patient_id: str) -> list[dict]:
        """Golden few-shot examples for the extraction prompt.

        Returns user-confirmed timeline entries from OTHER patients (the
        current patient's own confirmations are excluded) capped at
        ``GOLDEN_FEWSHOT_MAX_EXAMPLES``. Must never block the extraction:
        any failure — missing repo, disabled toggle, DB error — yields ``[]``.
        """
        if not GOLDEN_FEWSHOT_ENABLED:
            return []
        if self._timeline_repo is None:
            return []
        try:
            from ..extraction.golden_fewshot import select_examples

            golden = self._timeline_repo.get_golden()
            return select_examples(
                golden, patient_id, GOLDEN_FEWSHOT_MAX_EXAMPLES
            )
        except Exception:
            return []

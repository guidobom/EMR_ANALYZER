"""Clinical History Builder — builds the longitudinal clinical timeline.

Processes normalized clinical texts one at a time through the Clinical State
LLM, building an incremental temporal registry.  Afterwards a global
deduplication pass merges observations that describe the same clinical fact
with different wording.
"""

import difflib
import itertools
import json
import threading
from pathlib import Path
from typing import Callable, Optional

from ..config import active_workspace
from ..models.clinical_timeline import ClinicalTimelineEntry


class ClinicalHistoryBuilder:
    """Build a strictly temporal clinical history from normalized texts."""

    def __init__(
        self,
        timeline_repo,
        document_repo,
        cs_repo,
        clinical_state_llm_client,
        audit_repo=None,
    ):
        self._timeline_repo = timeline_repo
        self._doc_repo = document_repo
        self._cs_repo = cs_repo
        self._llm = clinical_state_llm_client
        self._audit = audit_repo

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_from_documents(
        self,
        patient_id: str,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        generate_narrative: bool = False,
    ) -> dict:
        """Run the full pipeline and persist the result.

        If *generate_narrative* is True, also generate the narrative
        clinical profile via LLM and store it in ClinicalState.

        Returns a summary dict with keys *total_entries*, *deduplicated*,
        and *final_entries*.
        """
        if not self._llm or not self._llm.is_available:
            raise RuntimeError(
                "Il modello LLM per il Clinical State non e' disponibile. "
                "Configuralo in Strumenti → Configura LLM."
            )

        # ---- 1. Load normalized texts sorted by document date ----------
        normalized_docs = self._load_normalized_docs(patient_id)
        if not normalized_docs:
            raise RuntimeError(
                "Nessun testo clinico normalizzato disponibile. "
                "Eseguire prima l'elaborazione dei documenti "
                "(Isolamento testo clinico)."
            )

        total = len(normalized_docs)

        # ---- 2. Per-document extraction with incremental registry ------
        all_entries: list[ClinicalTimelineEntry] = []
        registry_entries: list[dict] = []
        failed_docs: list[str] = []
        doc_stats: list[dict] = []

        # Local ID counter — avoids querying the DB for every entry
        # while entries are only saved at the end via save_batch().
        start_id_num = int(
            self._timeline_repo.get_next_entry_id().split("_")[1]
        )
        _id_seq = itertools.count(start_id_num)

        for i, ndoc in enumerate(normalized_docs):
            if progress_callback:
                pct = int((i / total) * 60)  # 0–60 % for per-doc phase
                progress_callback(
                    pct,
                    f"Analisi documento {i + 1}/{total}: {ndoc['doc_id']}",
                )

            registry_summary = self._format_registry_context(registry_entries)

            result = self._extract_for_document(ndoc, registry_summary)

            doc_entry_count = 0
            for entry_data in result.get("entries", []):
                entry_id = f"CTL_{next(_id_seq):06d}"
                entry = ClinicalTimelineEntry(
                    entry_id=entry_id,
                    patient_id=patient_id,
                    date_observed=self._normalize_date_observed(
                        entry_data.get("date_observed"),
                        ndoc["document_date"] or "",
                    ),
                    date_resolved=entry_data.get("date_resolved"),
                    category=entry_data.get("category", "other"),
                    description=entry_data.get("description", ""),
                    source_document_ids=[ndoc["doc_id"]],
                    source_texts=[entry_data.get("source_text", "")],
                    status=entry_data.get("status", "active"),
                    confidence=entry_data.get("confidence", 0.5),
                )
                all_entries.append(entry)
                registry_entries.append(entry.to_dict())
                doc_entry_count += 1

            if doc_entry_count == 0:
                failed_docs.append(ndoc["doc_id"])
            doc_stats.append({
                "doc_id": ndoc["doc_id"],
                "entries": doc_entry_count,
                "date": ndoc["document_date"],
            })

        if not all_entries:
            return {
                "total_entries": 0,
                "deduplicated": 0,
                "final_entries": 0,
            }

        # ---- 3. Merge lab entries + LLM deduplication --------------------
        lab_entries = self._lab_entries(patient_id)
        if lab_entries:
            for le in lab_entries:
                le.entry_id = f"CTL_{next(_id_seq):06d}"
            all_entries.extend(lab_entries)

        if progress_callback:
            progress_callback(65, f"Deduplicazione semantica LLM ({len(all_entries)} voci)...")

        final_entries = self._deduplicate_all(all_entries)
        removed_count = len(all_entries) - len(final_entries)

        # ---- 4. Persist -------------------------------------------------
        if progress_callback:
            progress_callback(80, "Salvataggio registro temporale...")

        self._timeline_repo.replace_all_for_patient(patient_id, final_entries)

        # ---- 5. Optionally generate narrative clinical_profile -----------
        if generate_narrative:
            if progress_callback:
                progress_callback(90, "Generazione profilo clinico narrativo...")
            narrative = self.generate_narrative(patient_id)
        else:
            narrative = None

        if progress_callback:
            progress_callback(100, "Storia clinica completata")

        # ---- 6. Audit ---------------------------------------------------
        if self._audit:
            self._audit.log(
                patient_id, "clinical_history_built", "timeline",
                patient_id,
                {
                    "documents_processed": total,
                    "extracted_entries": len(all_entries),
                    "deduplicated_entries": removed_count,
                    "final_entries": len(final_entries),
                },
            )

        return {
            "total_entries": len(all_entries),
            "deduplicated": removed_count,
            "final_entries": len(final_entries),
            "documents_processed": total,
            "documents_failed": len(failed_docs),
            "failed_doc_ids": failed_docs,
            "doc_stats": doc_stats,
        }

    def build_incremental(
        self,
        patient_id: str,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        generate_narrative: bool = False,
    ) -> dict:
        """Process only NEW documents and merge with the existing registry.

        Documents whose IDs already appear in the timeline are skipped.
        New entries are appended, then the FULL registry is deduplicated.
        The save is atomic: old entries are only removed after the new
        batch has been successfully saved.
        """
        if not self._llm or not self._llm.is_available:
            raise RuntimeError(
                "Il modello LLM per il Clinical State non e' disponibile."
            )

        # ---- 1. Find which documents are already in the registry ---------
        existing_entries = self._timeline_repo.get_by_patient(patient_id)
        existing_ids: set[str] = set()
        for e in existing_entries:
            existing_ids.update(e.source_document_ids)

        # ---- 2. Load only NEW normalized texts ---------------------------
        all_docs = self._load_normalized_docs(patient_id)
        new_docs = [d for d in all_docs if d["doc_id"] not in existing_ids]

        if not new_docs:
            return {
                "total_entries": 0, "deduplicated": 0,
                "final_entries": len(existing_entries),
                "documents_processed": 0,
                "documents_skipped": len(all_docs),
                "documents_failed": 0,
                "failed_doc_ids": [],
                "incremental": True,
            }

        total_new = len(new_docs)
        total_all = len(all_docs)

        # ---- 3. Extract from new documents only --------------------------
        new_entries: list[ClinicalTimelineEntry] = []
        failed_docs: list[str] = []

        start_id_num = int(
            self._timeline_repo.get_next_entry_id().split("_")[1]
        )
        _id_seq = itertools.count(start_id_num)

        for i, ndoc in enumerate(new_docs):
            if progress_callback:
                pct = int((i / total_new) * 50)
                progress_callback(
                    pct,
                    f"Nuovo doc {i + 1}/{total_new}: {ndoc['doc_id']} "
                    f"({total_all - total_new} saltati)",
                )

            result = self._extract_for_document(ndoc, "")

            for entry_data in result.get("entries", []):
                entry_id = f"CTL_{next(_id_seq):06d}"
                entry = ClinicalTimelineEntry(
                    entry_id=entry_id,
                    patient_id=patient_id,
                    date_observed=self._normalize_date_observed(
                        entry_data.get("date_observed"),
                        ndoc["document_date"] or "",
                    ),
                    date_resolved=entry_data.get("date_resolved"),
                    category=entry_data.get("category", "other"),
                    description=entry_data.get("description", ""),
                    source_document_ids=[ndoc["doc_id"]],
                    source_texts=[entry_data.get("source_text", "")],
                    status=entry_data.get("status", "active"),
                    confidence=entry_data.get("confidence", 0.5),
                )
                new_entries.append(entry)

            if not result.get("entries"):
                failed_docs.append(ndoc["doc_id"])

        if not new_entries and not existing_entries:
            return {
                "total_entries": 0, "deduplicated": 0, "final_entries": 0,
                "incremental": True,
            }

        # ---- 4. Merge old + new + lab ------------------------------------
        all_entries = existing_entries + new_entries
        lab_entries = self._lab_entries(patient_id)
        if lab_entries:
            for le in lab_entries:
                le.entry_id = f"CTL_{next(_id_seq):06d}"
            all_entries.extend(lab_entries)

        # ---- 5. Dedup the full merged registry --------------------------
        if progress_callback:
            progress_callback(
                60,
                f"Deduplica ({len(existing_entries)} esistenti + "
                f"{len(new_entries)} nuovi = {len(all_entries)} totali)...",
            )

        final_entries = self._deduplicate_all(all_entries)
        removed_count = len(all_entries) - len(final_entries)

        # ---- 6. Atomic save (delete old → save new in one shot) ---------
        if progress_callback:
            progress_callback(80, "Salvataggio atomico...")

        self._timeline_repo.replace_all_for_patient(patient_id, final_entries)

        # ---- 7. Optionally generate narrative ---------------------------
        if generate_narrative:
            if progress_callback:
                progress_callback(90, "Generazione profilo narrativo...")
            self.generate_narrative(patient_id)

        if progress_callback:
            progress_callback(100, "Storia clinica aggiornata")

        if self._audit:
            self._audit.log(
                patient_id, "clinical_history_incremental", "timeline",
                patient_id,
                {
                    "existing_entries": len(existing_entries),
                    "new_documents": total_new,
                    "new_entries": len(new_entries),
                    "deduplicated_entries": removed_count,
                    "final_entries": len(final_entries),
                },
            )

        return {
            "total_entries": len(new_entries),
            "deduplicated": removed_count,
            "final_entries": len(final_entries),
            "documents_processed": total_new,
            "documents_skipped": total_all - total_new,
            "documents_failed": len(failed_docs),
            "failed_doc_ids": failed_docs,
            "incremental": True,
        }

    def build_from_documents_parallel(
        self,
        patient_id: str,
        num_workers: int = 4,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        generate_narrative: bool = False,
    ) -> dict:
        """Parallel version — extracts from every document concurrently.

        Each document is processed **independently** (empty registry context)
        so the LLM calls can run in parallel.  A global deduplication pass
        afterwards merges observations that describe the same clinical fact.

        Returns the same summary dict as :meth:`build_from_documents`.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not self._llm or not self._llm.is_available:
            raise RuntimeError(
                "Il modello LLM per il Clinical State non e' disponibile. "
                "Configuralo in Strumenti → Configura LLM."
            )

        # ---- 1. Load normalized texts ------------------------------------
        normalized_docs = self._load_normalized_docs(patient_id)
        if not normalized_docs:
            raise RuntimeError(
                "Nessun testo clinico normalizzato disponibile."
            )
        total = len(normalized_docs)
        actual_workers = min(num_workers, total)

        if progress_callback:
            progress_callback(
                0,
                f"Analisi parallela di {total} documenti "
                f"({actual_workers} worker)...",
            )

        # ---- 2. Parallel extraction --------------------------------------
        all_entries: list[ClinicalTimelineEntry] = []
        failed_docs: list[str] = []
        doc_stats: list[dict] = []
        completed = 0
        t0 = time.monotonic()

        # Thread-safe entry-ID counter — avoids the race condition of
        # calling get_next_entry_id() from multiple threads (which all
        # see the same "current max" in the DB).
        start_id_num = int(
            self._timeline_repo.get_next_entry_id().split("_")[1]
        )
        _id_counter = itertools.count(start_id_num)
        _id_lock = threading.Lock()

        def _next_entry_id() -> str:
            with _id_lock:
                return f"CTL_{next(_id_counter):06d}"

        def _extract_one(ndoc: dict) -> tuple[str, list[dict], str | None]:
            """Extract entries from a single document (no registry context)."""
            result = self._extract_for_document(ndoc, "")
            return ndoc["doc_id"], result.get("entries", []), None

        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = {
                executor.submit(_extract_one, ndoc): ndoc
                for ndoc in normalized_docs
            }
            for future in as_completed(futures):
                completed += 1
                ndoc = futures[future]
                try:
                    doc_id, raw_entries, _ = future.result()
                except Exception:
                    failed_docs.append(ndoc["doc_id"])
                    doc_stats.append({
                        "doc_id": ndoc["doc_id"],
                        "entries": 0,
                        "date": ndoc["document_date"],
                        "error": "worker exception",
                    })
                    continue

                doc_entry_count = 0
                for entry_data in raw_entries:
                    entry_id = _next_entry_id()
                    entry = ClinicalTimelineEntry(
                        entry_id=entry_id,
                        patient_id=patient_id,
                        date_observed=self._normalize_date_observed(
                            entry_data.get("date_observed"),
                            ndoc["document_date"] or "",
                        ),
                        date_resolved=entry_data.get("date_resolved"),
                        category=entry_data.get("category", "other"),
                        description=entry_data.get("description", ""),
                        source_document_ids=[ndoc["doc_id"]],
                        source_texts=[entry_data.get("source_text", "")],
                        status=entry_data.get("status", "active"),
                        confidence=entry_data.get("confidence", 0.5),
                    )
                    all_entries.append(entry)
                    doc_entry_count += 1

                if doc_entry_count == 0:
                    failed_docs.append(ndoc["doc_id"])
                doc_stats.append({
                    "doc_id": ndoc["doc_id"],
                    "entries": doc_entry_count,
                    "date": ndoc["document_date"],
                })

                if progress_callback:
                    elapsed = time.monotonic() - t0
                    pct = int((completed / total) * 60)
                    progress_callback(
                        pct,
                        f"Completati {completed}/{total} documenti "
                        f"({elapsed:.0f}s)",
                    )

        if not all_entries:
            return {
                "total_entries": 0, "deduplicated": 0, "final_entries": 0,
                "documents_processed": total,
                "documents_failed": len(failed_docs),
                "failed_doc_ids": failed_docs, "doc_stats": doc_stats,
            }

        # ---- 3. Merge lab entries + LLM deduplication --------------------
        lab_entries = self._lab_entries(patient_id)
        if lab_entries:
            for le in lab_entries:
                le.entry_id = _next_entry_id()
            all_entries.extend(lab_entries)

        if progress_callback:
            progress_callback(65, f"Deduplicazione semantica LLM ({len(all_entries)} voci)...")

        final_entries = self._deduplicate_all(all_entries)
        removed_count = len(all_entries) - len(final_entries)

        # ---- 4. Persist ---------------------------------------------------
        if progress_callback:
            progress_callback(80, "Salvataggio registro temporale...")

        self._timeline_repo.replace_all_for_patient(patient_id, final_entries)

        # ---- 5. Optionally generate narrative -----------------------------
        if generate_narrative:
            if progress_callback:
                progress_callback(90, "Generazione profilo narrativo...")
            self.generate_narrative(patient_id)

        if progress_callback:
            progress_callback(100, "Storia clinica completata")

        elapsed = time.monotonic() - t0
        if self._audit:
            self._audit.log(
                patient_id, "clinical_history_built_parallel", "timeline",
                patient_id,
                {
                    "documents_processed": total,
                    "num_workers": actual_workers,
                    "extracted_entries": len(all_entries),
                    "deduplicated_entries": removed_count,
                    "final_entries": len(final_entries),
                    "elapsed_seconds": round(elapsed, 1),
                },
            )

        return {
            "total_entries": len(all_entries),
            "deduplicated": removed_count,
            "final_entries": len(final_entries),
            "documents_processed": total,
            "documents_failed": len(failed_docs),
            "failed_doc_ids": failed_docs,
            "doc_stats": doc_stats,
            "elapsed_seconds": round(elapsed, 1),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_normalized_docs(
        self, patient_id: str, skip_lab: bool = True
    ) -> list[dict]:
        """Return normalized texts sorted by document date (oldest first).

        If *skip_lab* is True, documents classified as ``laboratorio`` are
        excluded — their out-of-range values are converted to timeline
        entries deterministically via :meth:`_lab_entries`.
        """
        documents = self._doc_repo.list_by_patient(patient_id)
        documents = sorted(
            documents,
            key=lambda d: d.document_date or "9999-99-99",
        )
        normalized = []
        for doc in documents:
            if skip_lab and doc.document_type == "laboratorio":
                continue
            md_path = self._get_normalized_text_path(patient_id, doc.id)
            if md_path.exists():
                normalized.append({
                    "doc_id": doc.id,
                    "document_date": doc.document_date,
                    "document_type": doc.document_type,
                    "text": md_path.read_text(encoding="utf-8"),
                })
        return normalized

    def _lab_entries(self, patient_id: str) -> list[ClinicalTimelineEntry]:
        """Convert out-of-range lab values to timeline entries.

        Only values with ``is_abnormal=1`` are included.  Each entry records
        the parameter name, numeric value, unit, reference range and flag
        so the registry captures clinically significant laboratory findings
        without an LLM call.
        """
        from ..database.lab_repo import LabRepository
        lab_repo = LabRepository(self._timeline_repo.db)
        lab_values = lab_repo.get_by_patient(patient_id)
        entries = []
        for lv in lab_values:
            if not lv.is_abnormal:
                continue
            ref = f" [{lv.reference_text}]" if lv.reference_text else ""
            flag = f" ({lv.flag})" if lv.flag else ""
            if lv.value_text:
                display_value = lv.value_text
            elif lv.value is not None:
                display_value = f"{lv.operator or ''}{lv.value} {lv.unit or ''}".strip()
            else:
                display_value = ""
            entries.append(ClinicalTimelineEntry(
                entry_id="",  # assigned later
                patient_id=patient_id,
                date_observed=lv.sample_date or "",
                category="laboratory",
                description=(
                    f"{lv.parameter_name}: {display_value}{ref}{flag}"
                ),
                source_document_ids=[lv.document_id],
                source_texts=[lv.source_text or ""],
                status="active",
                confidence=1.0,  # deterministic extraction → high confidence
            ))
        return entries

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
            system_prompt = (
                "Sei un medico che redige una storia clinica sintetica. "
                "Basandoti ESCLUSIVAMENTE sulle osservazioni fornite, produci "
                "un profilo clinico narrativo in italiano, organizzato "
                "cronologicamente. Non aggiungere informazioni non presenti. "
                "Usa un linguaggio clinico professionale ma chiaro."
            )

            user_prompt = f"""Redigi un profilo clinico narrativo basato sulle seguenti osservazioni cliniche in ordine cronologico.

Non inventare nulla. Se un dato non e' presente, non menzionarlo.
Organizza il testo in paragrafi cronologici coerenti.
Cita le date quando disponibili.

OSSERVAZIONI CLINICHE:
{json.dumps(entries_data, ensure_ascii=False, indent=2)}
"""
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

    # Conservative threshold: only near-verbatim entries (same date +
    # category, ≥0.75 similarity) are merged deterministically.  The LLM
    # stage handles the semantic duplicates the string matcher misses.
    _DETERMINISTIC_DEDUP_THRESHOLD = 0.75

    @staticmethod
    def _deterministic_dedup(
        entries: list[ClinicalTimelineEntry],
    ) -> list[ClinicalTimelineEntry]:
        """Merge near-verbatim entries with same date + category.

        Runs before the LLM dedup to reduce the entry count and avoid
        context-window overflow.  Uses ``SequenceMatcher`` with a
        conservative threshold (0.75) so only entries that express the
        same clinical fact with trivial wording differences are merged
        (e.g. \"Inizio dabrafenib 150 mg\" vs \"inizia dabrafenib 150 mg\").
        The survivor is the longest / highest-confidence entry of the group;
        it accumulates the ``merged_into_ids`` (excluding its own id) and
        the sources of the fused entries.
        """
        if len(entries) <= 1:
            return entries

        threshold = ClinicalHistoryBuilder._DETERMINISTIC_DEDUP_THRESHOLD

        # Sort by (date, category, description) for stable grouping
        sorted_entries = sorted(
            entries,
            key=lambda e: (e.date_observed, e.category, e.description),
        )

        merged: list[ClinicalTimelineEntry] = []
        used: set[int] = set()

        for i, ei in enumerate(sorted_entries):
            if i in used:
                continue

            # Collect the group: same date + category, near-verbatim text.
            group: list[int] = [i]
            for j, ej in enumerate(sorted_entries):
                if j <= i or j in used:
                    continue
                if ei.date_observed != ej.date_observed:
                    break  # sorted — no more same-date entries
                if ei.category != ej.category:
                    continue
                ratio = difflib.SequenceMatcher(
                    None, ei.description.lower(), ej.description.lower()
                ).ratio()
                if ratio >= threshold:
                    group.append(j)
                    used.add(j)

            # Survivor: longest description, then highest confidence.
            best_idx = max(
                group,
                key=lambda idx: (
                    len(sorted_entries[idx].description),
                    sorted_entries[idx].confidence,
                ),
            )
            survivor = sorted_entries[best_idx]

            if len(group) > 1:
                merged_ids = [
                    sorted_entries[idx].entry_id
                    for idx in group if idx != best_idx
                ]
                all_doc_ids = list(survivor.source_document_ids)
                all_texts = list(survivor.source_texts)
                for idx in group:
                    if idx == best_idx:
                        continue
                    me = sorted_entries[idx]
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
                    max(sorted_entries[idx].confidence for idx in group),
                )
                survivor.merged_into_ids = list(dict.fromkeys(
                    survivor.merged_into_ids + merged_ids
                ))

            merged.append(survivor)
            used.add(i)

        return merged

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

    def _extract_for_document(
        self,
        ndoc: dict,
        registry_summary: str,
    ) -> dict:
        """Route to the best extraction method based on document type.

        Discharge letters from pre-acute / post-acute wards get a
        specialised prompt that understands their clinical structure.
        All other documents use the generic timeline extractor.
        """
        from ..pipeline.classifier import DocumentClassifier

        is_discharge = ndoc.get("document_type") == "lettera_dimissione"
        is_pre_acute = DocumentClassifier.is_pre_acute_discharge(
            ndoc.get("text", "")
        )

        if is_discharge and is_pre_acute:
            return self._llm.extract_from_discharge_letter(
                ndoc["text"],
                ndoc.get("document_date"),
                registry_summary,
            )
        elif is_discharge:
            # Standard discharge letter — still benefits from the
            # discharge-aware prompt.
            return self._llm.extract_from_discharge_letter(
                ndoc["text"],
                ndoc.get("document_date"),
                registry_summary,
            )
        else:
            return self._llm.extract_timeline_entries(
                ndoc["text"],
                registry_summary,
                ndoc.get("document_date"),
            )

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

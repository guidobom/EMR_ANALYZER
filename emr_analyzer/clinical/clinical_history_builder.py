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

from ..config import (
    active_workspace,
    GOLDEN_FEWSHOT_ENABLED,
    GOLDEN_FEWSHOT_MAX_EXAMPLES,
)
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
        if not self._llm or not self._llm.is_available:
            raise RuntimeError(
                "Il modello LLM per il Clinical State non e' disponibile. "
                "Configuralo in Strumenti → Configura LLM."
            )

        # Few-shot style examples from the golden set (OTHER patients),
        # fetched once for the whole run.
        golden_examples = self._load_golden_examples(patient_id)

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

            result = self._extract_for_document(
                ndoc, registry_summary, golden_examples
            )

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
        if not self._llm or not self._llm.is_available:
            raise RuntimeError(
                "Il modello LLM per il Clinical State non e' disponibile."
            )

        # ---- 1. Find which documents are already in the registry ---------
        existing_entries = self._timeline_repo.get_by_patient(patient_id)
        existing_ids: set[str] = set()
        for e in existing_entries:
            existing_ids.update(e.source_document_ids)

        # Few-shot style examples from the golden set (OTHER patients).
        golden_examples = self._load_golden_examples(patient_id)

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

            result = self._extract_for_document(
                ndoc, "", golden_examples
            )

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

        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not self._llm or not self._llm.is_available:
            raise RuntimeError(
                "Il modello LLM per il Clinical State non e' disponibile. "
                "Configuralo in Strumenti → Configura LLM."
            )

        # Few-shot style examples from the golden set (OTHER patients).
        # Fetched once before the pool and only read by the workers
        # (a plain list of dicts — thread-safe).
        golden_examples = self._load_golden_examples(patient_id)

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
            result = self._extract_for_document(
                ndoc, "", golden_examples
            )
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
        if self._registry_builder is not None:
            before = self._timeline_repo.count_by_patient(patient_id)
            result = self._registry_builder.build(
                patient_id, incremental=True, num_workers=1
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

    def _extract_for_document(
        self,
        ndoc: dict,
        registry_summary: str,
        golden_examples: Optional[list[dict]] = None,
    ) -> dict:
        """Route to the best extraction method based on document type.

        Discharge letters from pre-acute / post-acute wards get a
        specialised prompt that understands their clinical structure.
        All other documents use the generic timeline extractor.

        *golden_examples* are passed as keyword arguments because the two
        LLM methods use a different positional argument order.
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
                golden_examples=golden_examples,
            )
        elif is_discharge:
            # Standard discharge letter — still benefits from the
            # discharge-aware prompt.
            return self._llm.extract_from_discharge_letter(
                ndoc["text"],
                ndoc.get("document_date"),
                registry_summary,
                golden_examples=golden_examples,
            )
        else:
            return self._llm.extract_timeline_entries(
                ndoc["text"],
                registry_summary,
                ndoc.get("document_date"),
                golden_examples=golden_examples,
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

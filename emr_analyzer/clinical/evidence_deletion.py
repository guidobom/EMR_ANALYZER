"""Reset of a patient's atomic evidence and everything derived from it.

Deletes the atomic evidence layer (``clinical_evidence`` plus its cascade
side tables), the chronological registry, the narrative profile and the
patient's gold set annotations.  What remains: documents, structured
laboratory values, routing identity and the audit trail (plus the new audit
entry).

Registry events are rejected through the review flow — not deleted — so the
reset keeps a full audit history, exactly like the "Elimina Registro" GUI
action.  Repositories that self-commit cannot participate in the single
transaction, so the service uses direct SQL inside ``with self.db:`` (the
same style as ``DocumentDeletionService``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json


# Queue item types whose targets die with the evidence rows.  Event queue
# rows survive, in parity with "Elimina Registro": the next validation
# preparation re-syncs them.
_EVIDENCE_QUEUE_ITEM_TYPES = (
    "evidence", "atomic_duplicate_v3", "evidence_relation_v3",
)

# Gold tables with a patient_id column, deleted by explicit patient sweep
# (cascades would cover decisions and sources, but the sweep is auditable).
_GOLD_TABLES = (
    "gold_adjudication_decisions", "gold_annotation_sources",
    "gold_annotations", "gold_atomic_annotations", "gold_set_cases",
)


@dataclass
class EvidenceDeletionResult:
    patient_id: str
    deleted: bool = False
    removed_evidence_count: int = 0
    rejected_event_count: int = 0
    timeline_entries_removed: int = 0
    profile_cleared: bool = False
    validation_queue_removed: int = 0
    excluded_evidence_removed: int = 0
    gold_rows_removed: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


class EvidenceDeletionService:
    """Delete the full derived stack of one patient, atomically."""

    def __init__(self, db, review_repo, registry_repo, cs_repo,
                 audit_repo=None):
        self.db = db
        self.review_repo = review_repo
        self.registry_repo = registry_repo
        self.cs_repo = cs_repo
        self.audit_repo = audit_repo

    def delete(self, patient_id: str) -> EvidenceDeletionResult:
        result = EvidenceDeletionResult(patient_id=patient_id)

        evidence_ids = [
            row["evidence_id"] for row in self.db.execute(
                "SELECT evidence_id FROM clinical_evidence WHERE patient_id=?",
                (patient_id,),
            ).fetchall()
        ]
        events = self.registry_repo.get_events(patient_id)
        result.timeline_entries_removed = self.db.execute(
            "SELECT COUNT(*) FROM clinical_timeline WHERE patient_id=?",
            (patient_id,),
        ).fetchone()[0]

        state = self.cs_repo.load(patient_id) if self.cs_repo else None
        state_json = None
        if state is not None:
            state.clinical_profile = ""
            state_json = json.dumps(state.to_dict(), ensure_ascii=False)
            result.profile_cleared = True

        try:
            with self.db:
                self._delete_database_records(
                    patient_id, events, state_json, result
                )
            result.deleted = True
            result.removed_evidence_count = len(evidence_ids)
            result.rejected_event_count = len(events)
        except Exception as exc:
            result.error = str(exc)
            return result

        if self.audit_repo:
            try:
                self.audit_repo.log(
                    patient_id,
                    "delete",
                    "evidence",
                    patient_id,
                    {
                        "removed_evidence_count": len(evidence_ids),
                        "rejected_event_count": len(events),
                        "timeline_entries_removed": (
                            result.timeline_entries_removed
                        ),
                        "profile_cleared": result.profile_cleared,
                        "validation_queue_removed": (
                            result.validation_queue_removed
                        ),
                        "excluded_evidence_removed": (
                            result.excluded_evidence_removed
                        ),
                        "gold_rows_removed": result.gold_rows_removed,
                    },
                )
            except Exception as exc:
                result.warnings.append(f"Audit log non aggiornato: {exc}")
        return result

    def _delete_database_records(self, patient_id, events, state_json,
                                 result) -> None:
        """Delete the derived stack.  Runs inside the caller's transaction.

        Order matters: queue and claim-source rows reference targets that
        the ``clinical_evidence`` delete removes, and the claim-source
        subquery must run before that delete.
        """
        cursor = self.db.execute(
            """DELETE FROM validation_queue
               WHERE patient_id=? AND item_type IN (?, ?, ?)""",
            (patient_id, *_EVIDENCE_QUEUE_ITEM_TYPES),
        )
        result.validation_queue_removed = cursor.rowcount

        cursor = self.db.execute(
            """DELETE FROM clinical_event_claim_sources
               WHERE source_type='evidence'
                 AND source_id IN (
                     SELECT evidence_id FROM clinical_evidence
                     WHERE patient_id=?)""",
            (patient_id,),
        )

        cursor = self.db.execute(
            "DELETE FROM excluded_evidence WHERE patient_id=?", (patient_id,)
        )
        result.excluded_evidence_removed = cursor.rowcount

        for table in _GOLD_TABLES:
            cursor = self.db.execute(
                f"DELETE FROM {table} WHERE patient_id=?", (patient_id,)
            )
            result.gold_rows_removed += cursor.rowcount

        # Without this the pipeline status would still report the atomic
        # phase as current and re-extraction would silently do nothing.
        self.db.execute(
            """DELETE FROM processing_manifest
               WHERE patient_id=? AND stage='atomic_evidence'""",
            (patient_id,),
        )

        # FK cascades remove evidence_source_refs, duplicate groups and
        # members, relations, the adjudication cache, aggregation coverage
        # and clinical_event_evidence links.
        self.db.execute(
            "DELETE FROM clinical_evidence WHERE patient_id=?", (patient_id,)
        )

        for event in events:
            self.review_repo.decide_event(
                patient_id,
                event.event_id,
                "rejected",
                reason="Esclusione richiesta con azzeramento di evidenze e registro",
            )

        # TimelineRepository.delete_by_patient self-commits: direct SQL here.
        self.db.execute(
            "DELETE FROM clinical_timeline WHERE patient_id=?", (patient_id,)
        )

        # ClinicalStateRepository.save self-commits: direct SQL here, with
        # the same version bump the repository would apply.
        if state_json is not None:
            self.db.execute(
                """UPDATE clinical_state
                   SET state_json=?, updated_at=?, version=version+1
                   WHERE patient_id=?""",
                (state_json, datetime.now().isoformat(), patient_id),
            )

"""Persistent human review decisions that survive registry rebuilds."""

from __future__ import annotations

import json

from .engine import DatabaseEngine
from ..models.clinical_registry import ReviewDecision


class ReviewDecisionRepository:
    def __init__(self, db: DatabaseEngine):
        self.db = db

    def add(self, decision: ReviewDecision) -> None:
        with self.db:
            self.db.execute(
                """INSERT INTO review_decisions
                   (decision_id, patient_id, target_type, target_id, decision,
                    reviewer_id, reviewer_role, reason, previous_value_json,
                    corrected_value_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    decision.decision_id, decision.patient_id,
                    decision.target_type, decision.target_id,
                    decision.decision, decision.reviewer_id,
                    decision.reviewer_role, decision.reason,
                    json.dumps(decision.previous_value, ensure_ascii=False),
                    json.dumps(decision.corrected_value, ensure_ascii=False),
                    decision.created_at,
                ),
            )
            if decision.target_type == "clinical_event":
                allowed = {
                    "summary_short", "summary_detail", "category",
                    "canonical_entity", "anatomical_site", "laterality",
                    "severity", "significance", "status", "certainty",
                    "assertion", "first_evidence_date", "date_end",
                    "date_precision",
                }
                corrections = {
                    key: value for key, value in decision.corrected_value.items()
                    if key in allowed
                }
                if corrections:
                    assignments = ", ".join(f'"{key}"=?' for key in corrections)
                    self.db.execute(
                        f"UPDATE clinical_events SET {assignments} WHERE event_id=?",
                        (*corrections.values(), decision.target_id),
                    )
                self.db.execute(
                    """UPDATE clinical_events
                       SET review_status=?, version=version+1,
                           updated_at=? WHERE event_id=?""",
                    (decision.decision, decision.created_at, decision.target_id),
                )

    def decide_event(
        self,
        patient_id: str,
        event_id: str,
        decision: str,
        *,
        corrected_value: dict | None = None,
        reason: str = "",
        reviewer_id: str = "local_user",
        reviewer_role: str = "clinician",
    ) -> ReviewDecision:
        """Persist a complete, auditable event decision and apply corrections."""
        row = self.db.execute(
            "SELECT * FROM clinical_events WHERE event_id=? AND patient_id=?",
            (event_id, patient_id),
        ).fetchone()
        if row is None:
            raise ValueError(f"Evento clinico non trovato: {event_id}")
        previous = dict(row)
        previous["structured_data"] = json.loads(
            previous.pop("structured_data_json") or "{}"
        )
        item = ReviewDecision(
            patient_id=patient_id,
            target_type="clinical_event",
            target_id=event_id,
            decision=decision,
            reviewer_id=reviewer_id,
            reviewer_role=reviewer_role,
            reason=reason,
            previous_value=previous,
            corrected_value=corrected_value or {},
        )
        self.add(item)
        return item

    def get_for_target(self, target_type: str, target_id: str) -> list[dict]:
        rows = self.db.execute(
            """SELECT * FROM review_decisions
               WHERE target_type=? AND target_id=? ORDER BY created_at""",
            (target_type, target_id),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["previous_value"] = json.loads(
                item.pop("previous_value_json") or "{}"
            )
            item["corrected_value"] = json.loads(
                item.pop("corrected_value_json") or "{}"
            )
            result.append(item)
        return result

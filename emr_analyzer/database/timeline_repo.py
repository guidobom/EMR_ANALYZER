"""Clinical Timeline repository."""

import json
from datetime import datetime
from typing import Optional

from .engine import DatabaseEngine
from ..models.clinical_timeline import ClinicalTimelineEntry


class TimelineRepository:
    """Data access for clinical_timeline table."""

    def __init__(self, db: DatabaseEngine):
        self.db = db

    def save_batch(self, entries: list[ClinicalTimelineEntry]) -> None:
        """Insert or replace a batch of timeline entries."""
        if not entries:
            return
        now = datetime.now().isoformat()
        self.db.executemany(
            """INSERT OR REPLACE INTO clinical_timeline
               (entry_id, patient_id, date_observed, date_resolved, category,
                description, source_document_ids, source_texts,
                merged_into_ids, status, confidence, is_golden,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    e.entry_id, e.patient_id, e.date_observed, e.date_resolved,
                    e.category, e.description,
                    json.dumps(e.source_document_ids, ensure_ascii=False),
                    json.dumps(e.source_texts, ensure_ascii=False),
                    json.dumps(e.merged_into_ids, ensure_ascii=False),
                    e.status, e.confidence, e.is_golden,
                    e.created_at or now, now,
                )
                for e in entries
            ],
        )
        self.db.commit()

    def get_by_patient(self, patient_id: str) -> list[ClinicalTimelineEntry]:
        """Get all entries for a patient, chronologically ordered."""
        cursor = self.db.execute(
            """SELECT * FROM clinical_timeline
               WHERE patient_id=? ORDER BY date_observed ASC, entry_id ASC""",
            (patient_id,),
        )
        return [self._row_to_entry(r) for r in cursor.fetchall()]

    def delete_by_patient(self, patient_id: str) -> None:
        """Remove all timeline entries for a patient."""
        self.db.execute(
            "DELETE FROM clinical_timeline WHERE patient_id=?", (patient_id,)
        )
        self.db.commit()

    def replace_all_for_patient(
        self, patient_id: str, entries: list[ClinicalTimelineEntry]
    ) -> None:
        """Atomically replace all entries for a patient with a new batch."""
        with self.db:
            self.db.execute(
                "DELETE FROM clinical_timeline WHERE patient_id=?",
                (patient_id,),
            )
            if entries:
                now = datetime.now().isoformat()
                self.db.executemany(
                    """INSERT INTO clinical_timeline
                       (entry_id, patient_id, date_observed, date_resolved,
                        category, description, source_document_ids,
                        source_texts, merged_into_ids, status, confidence,
                        is_golden, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            e.entry_id, e.patient_id, e.date_observed,
                            e.date_resolved, e.category, e.description,
                            json.dumps(e.source_document_ids, ensure_ascii=False),
                            json.dumps(e.source_texts, ensure_ascii=False),
                            json.dumps(e.merged_into_ids, ensure_ascii=False),
                            e.status, e.confidence, e.is_golden,
                            e.created_at or now, now,
                        )
                        for e in entries
                    ],
                )

    def count_by_patient(self, patient_id: str) -> int:
        """Return the number of timeline entries for a patient."""
        cursor = self.db.execute(
            "SELECT COUNT(*) FROM clinical_timeline WHERE patient_id=?",
            (patient_id,),
        )
        return cursor.fetchone()[0]

    def delete_entry(self, entry_id: str) -> None:
        """Delete a single timeline entry by its ID."""
        self.db.execute(
            "DELETE FROM clinical_timeline WHERE entry_id=?", (entry_id,)
        )
        self.db.commit()

    def get_golden(
        self, patient_id: Optional[str] = None
    ) -> list[ClinicalTimelineEntry]:
        """Return the user-confirmed golden entries, optionally scoped to a
        single patient."""
        if patient_id is not None:
            cursor = self.db.execute(
                """SELECT * FROM clinical_timeline
                   WHERE is_golden=1 AND patient_id=?
                   ORDER BY date_observed ASC, entry_id ASC""",
                (patient_id,),
            )
        else:
            cursor = self.db.execute(
                """SELECT * FROM clinical_timeline
                   WHERE is_golden=1
                   ORDER BY patient_id ASC, date_observed ASC, entry_id ASC"""
            )
        return [self._row_to_entry(r) for r in cursor.fetchall()]

    def set_golden(self, entry_id: str, is_golden: bool) -> None:
        """Mark a timeline entry as user-confirmed golden (or unmark it)."""
        now = datetime.now().isoformat()
        self.db.execute(
            "UPDATE clinical_timeline SET is_golden=?, updated_at=? "
            "WHERE entry_id=?",
            (1 if is_golden else 0, now, entry_id),
        )
        self.db.commit()

    def update_description(self, entry_id: str, description: str) -> None:
        """Set a user-edited canonical description.

        Editing a description implies a human confirmation, so the entry is
        also marked as golden.
        """
        now = datetime.now().isoformat()
        self.db.execute(
            "UPDATE clinical_timeline SET description=?, is_golden=1, "
            "updated_at=? WHERE entry_id=?",
            (description, now, entry_id),
        )
        self.db.commit()

    def get_next_entry_id(self) -> str:
        """Generate the next sequential timeline entry ID."""
        cursor = self.db.execute(
            "SELECT entry_id FROM clinical_timeline ORDER BY entry_id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if row:
            last_id = row["entry_id"]
            last_num = int(last_id.split("_")[1])
            return f"CTL_{last_num + 1:06d}"
        return "CTL_000001"

    @staticmethod
    def _row_to_entry(row) -> ClinicalTimelineEntry:
        return ClinicalTimelineEntry(
            entry_id=row["entry_id"],
            patient_id=row["patient_id"],
            date_observed=row["date_observed"],
            date_resolved=row["date_resolved"],
            category=row["category"],
            description=row["description"],
            source_document_ids=json.loads(row["source_document_ids"] or "[]"),
            source_texts=json.loads(row["source_texts"] or "[]"),
            merged_into_ids=json.loads(row["merged_into_ids"] or "[]"),
            status=row["status"],
            confidence=row["confidence"] or 0.5,
            is_golden=row["is_golden"] or 0,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

"""Clinical Event repository — CRUD for the Event Store."""

from datetime import datetime
from typing import Optional

from .engine import DatabaseEngine
from ..models.clinical_event import ClinicalEvent


class EventRepository:
    """Data access for the clinical Event Store."""

    def __init__(self, db: DatabaseEngine):
        self.db = db

    def insert(self, event: ClinicalEvent) -> None:
        self.db.execute(
            """INSERT INTO clinical_events (event_id, patient_id, event_date,
               event_type, entity, value, unit, status, source_document_id,
               page, source_text, confidence, validated_by_user, validated_at,
               user_notes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (event.event_id, event.patient_id, event.event_date,
             event.event_type, event.entity, event.value, event.unit,
             event.status, event.source_document_id, event.page,
             event.source_text, event.confidence,
             1 if event.validated_by_user else 0, event.validated_at,
             event.user_notes, event.created_at),
        )
        self.db.commit()

    def insert_batch(self, events: list[ClinicalEvent]) -> None:
        params = [
            (ev.event_id, ev.patient_id, ev.event_date, ev.event_type,
             ev.entity, ev.value, ev.unit, ev.status, ev.source_document_id,
             ev.page, ev.source_text, ev.confidence,
             1 if ev.validated_by_user else 0, ev.validated_at,
             ev.user_notes, ev.created_at)
            for ev in events
        ]
        self.db.executemany(
            """INSERT INTO clinical_events (event_id, patient_id, event_date,
               event_type, entity, value, unit, status, source_document_id,
               page, source_text, confidence, validated_by_user, validated_at,
               user_notes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            params,
        )
        self.db.commit()

    def get_by_patient(self, patient_id: str) -> list[ClinicalEvent]:
        cursor = self.db.execute(
            """SELECT * FROM clinical_events WHERE patient_id=?
               ORDER BY event_date ASC, created_at ASC""",
            (patient_id,),
        )
        return [self._row_to_event(r) for r in cursor.fetchall()]

    def get_by_document(self, document_id: str) -> list[ClinicalEvent]:
        cursor = self.db.execute(
            "SELECT * FROM clinical_events WHERE source_document_id=?",
            (document_id,),
        )
        return [self._row_to_event(r) for r in cursor.fetchall()]

    def get_by_type(self, patient_id: str, event_type: str) -> list[ClinicalEvent]:
        cursor = self.db.execute(
            """SELECT * FROM clinical_events
               WHERE patient_id=? AND event_type=?
               ORDER BY event_date ASC""",
            (patient_id, event_type),
        )
        return [self._row_to_event(r) for r in cursor.fetchall()]

    def get_by_date_range(self, patient_id: str, start: str,
                          end: str) -> list[ClinicalEvent]:
        cursor = self.db.execute(
            """SELECT * FROM clinical_events
               WHERE patient_id=? AND event_date BETWEEN ? AND ?
               ORDER BY event_date ASC""",
            (patient_id, start, end),
        )
        return [self._row_to_event(r) for r in cursor.fetchall()]

    def update_status(self, event_id: str, status: str,
                      notes: Optional[str] = None) -> None:
        self.db.execute(
            """UPDATE clinical_events SET status=?, validated_by_user=1,
               validated_at=?, user_notes=?
               WHERE event_id=?""",
            (status, datetime.now().isoformat(), notes, event_id),
        )
        self.db.commit()

    def update_entity(self, event_id: str, entity: str,
                      value: Optional[float] = None,
                      unit: Optional[str] = None,
                      event_date: Optional[str] = None) -> None:
        self.db.execute(
            """UPDATE clinical_events
               SET entity=?, value=?, unit=?, event_date=?, validated_by_user=1,
               validated_at=?
               WHERE event_id=?""",
            (entity, value, unit, event_date, datetime.now().isoformat(), event_id),
        )
        self.db.commit()

    def delete_by_document(self, document_id: str) -> None:
        self.db.execute(
            "DELETE FROM clinical_events WHERE source_document_id=?",
            (document_id,),
        )
        self.db.commit()

    def reject_event(self, event_id: str, notes: Optional[str] = None) -> None:
        self.update_status(event_id, "rejected", notes)

    def confirm_event(self, event_id: str, notes: Optional[str] = None) -> None:
        self.update_status(event_id, "confirmed", notes)

    def get_next_id(self) -> str:
        cursor = self.db.execute(
            "SELECT event_id FROM clinical_events ORDER BY event_id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if row:
            last_id = row["event_id"]
            last_num = int(last_id.split("_")[1])
            return f"EVT_{last_num + 1:06d}"
        return "EVT_000001"

    def count_by_patient(self, patient_id: str) -> int:
        cursor = self.db.execute(
            "SELECT COUNT(*) FROM clinical_events WHERE patient_id=?",
            (patient_id,),
        )
        return cursor.fetchone()[0]

    def _row_to_event(self, row) -> ClinicalEvent:
        return ClinicalEvent(
            event_id=row["event_id"],
            patient_id=row["patient_id"],
            event_date=row["event_date"],
            event_type=row["event_type"],
            entity=row["entity"],
            value=row["value"],
            unit=row["unit"],
            status=row["status"],
            source_document_id=row["source_document_id"],
            page=row["page"],
            source_text=row["source_text"] or "",
            confidence=row["confidence"] or 0.0,
            validated_by_user=bool(row["validated_by_user"]),
            validated_at=row["validated_at"],
            user_notes=row["user_notes"],
            created_at=row["created_at"],
        )

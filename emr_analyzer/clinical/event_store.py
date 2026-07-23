"""Event Store — manages clinical event persistence and queries."""

from datetime import datetime
from typing import Optional

from ..database.event_repo import EventRepository
from ..models.clinical_event import ClinicalEvent


class EventStore:
    """
    Manages the Event Store for a patient.
    Provides CRUD operations and event-based queries.
    """

    def __init__(self, event_repo: EventRepository,
                 audit_repo=None):
        self._repo = event_repo
        self._audit = audit_repo

    def add_events(self, events: list[ClinicalEvent]) -> list[ClinicalEvent]:
        """Add new events to the store, assigning proper IDs."""
        stored = []
        for event in events:
            event.event_id = self._repo.get_next_id()
            event.created_at = datetime.now().isoformat()
            self._repo.insert(event)
            stored.append(event)

            if self._audit:
                self._audit.log(
                    event.patient_id, "event_created",
                    "event", event.event_id,
                    {"event_type": event.event_type, "entity": event.entity},
                )

        return stored

    def get_patient_events(self, patient_id: str) -> list[ClinicalEvent]:
        return self._repo.get_by_patient(patient_id)

    def get_events_by_type(self, patient_id: str,
                           event_type: str) -> list[ClinicalEvent]:
        return self._repo.get_by_type(patient_id, event_type)

    def get_events_by_document(self, document_id: str) -> list[ClinicalEvent]:
        return self._repo.get_by_document(document_id)

    def confirm_event(self, event_id: str, notes: str = None):
        self._repo.confirm_event(event_id, notes)
        if self._audit:
            self._audit.log(
                "", "event_confirmed", "event", event_id,
                {"notes": notes},
            )

    def reject_event(self, event_id: str, notes: str = None):
        self._repo.reject_event(event_id, notes)

    def update_event(self, event_id: str, entity: str = None,
                     value: float = None, unit: str = None,
                     event_date: str = None):
        self._repo.update_entity(event_id, entity, value, unit, event_date)

    def build_timeline(self, patient_id: str) -> list[dict]:
        """Build a chronological timeline of all events."""
        events = self._repo.get_by_patient(patient_id)
        timeline = []

        for event in events:
            timeline.append({
                "date": event.event_date,
                "type": event.event_type,
                "entity": event.entity,
                "value": event.value,
                "unit": event.unit,
                "status": event.status,
                "event_id": event.event_id,
                "document_id": event.source_document_id,
                "page": event.page,
                "confidence": event.confidence,
                "source_text": event.source_text[:200] if event.source_text else "",
                "validated": event.validated_by_user,
            })

        return sorted(timeline, key=lambda e: e["date"])

    def get_statistics(self, patient_id: str) -> dict:
        """Get event statistics for a patient."""
        events = self._repo.get_by_patient(patient_id)

        types = {}
        confirmed = 0
        proposed = 0
        rejected = 0

        for e in events:
            types[e.event_type] = types.get(e.event_type, 0) + 1
            if e.status == "confirmed":
                confirmed += 1
            elif e.status == "rejected":
                rejected += 1
            else:
                proposed += 1

        return {
            "total": len(events),
            "by_type": types,
            "confirmed": confirmed,
            "proposed": proposed,
            "rejected": rejected,
        }

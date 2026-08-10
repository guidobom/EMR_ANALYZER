"""Clinical State Manager — builds and maintains the consolidated clinical picture."""

import json
from datetime import datetime
from typing import Optional

from ..models.clinical_state import (
    ClinicalState, ClinicalStateDelta,
    Diagnosis, Treatment, Toxicity, Procedure, Biomarker,
)
from ..models.clinical_event import ClinicalEvent
from ..database.clinical_state_repo import ClinicalStateRepository
from ..database.event_repo import EventRepository


class ClinicalStateManager:
    """
    Manages the Clinical State — a consolidated, deduplicated
    clinical picture of the patient.
    """

    def __init__(self, cs_repo: ClinicalStateRepository,
                 event_repo: EventRepository,
                 audit_repo=None,
                 llm_client=None,
                 evidence_repo=None,
                 projection_repo=None):
        self._cs_repo = cs_repo
        self._event_repo = event_repo
        self._audit = audit_repo
        self._llm_client = llm_client
        self._evidence_repo = evidence_repo
        self._projection_repo = projection_repo

    def get_or_create(self, patient_id: str) -> ClinicalState:
        """Get existing Clinical State or create a new empty one."""
        state = self._cs_repo.load(patient_id)
        if state is None:
            state = ClinicalState(patient_id=patient_id)
            self._cs_repo.save(state)
        return state

    def build_initial(self, patient_id: str,
                      events: list[ClinicalEvent] = None) -> ClinicalState:
        """Build initial Clinical State from all events."""
        if events is None:
            events = self._event_repo.get_by_patient(patient_id)

        state = self.get_or_create(patient_id)

        for event in events:
            if event.status == "rejected":
                continue
            self._apply_event_to_state(state, event)

        self._apply_patient_evidence(state, patient_id)
        self._apply_patient_projections(state, patient_id)

        self._cs_repo.save(state)
        return state

    def propose_delta(self, patient_id: str,
                      new_events: list[ClinicalEvent]) -> ClinicalStateDelta:
        """
        Propose a delta to the Clinical State based on new events.
        Uses the Clinical State LLM if available, otherwise applies deterministic rules.
        """
        current_state = self.get_or_create(patient_id)

        # Try LLM-based delta first
        if self._llm_client and self._llm_client.is_available:
            return self._llm_delta(current_state, new_events)

        # Deterministic fallback
        return self._deterministic_delta(current_state, new_events)

    def apply_delta(self, patient_id: str, delta: ClinicalStateDelta,
                    document_id: str = None,
                    auto_apply_low_risk: bool = True) -> ClinicalStateDelta:
        """
        Apply a ClinicalStateDelta to the patient's state.
        Returns the portion that needs human validation.
        """
        state = self.get_or_create(patient_id)
        needs_validation = ClinicalStateDelta()

        # Auto-apply items
        for item in delta.low_risk_changes:
            self._apply_delta_item(state, "confirm", item)

        # Also auto-apply add/update items with high confidence
        for item in delta.add + delta.update:
            self._apply_delta_item(state, "add", item)
            # Mark as auto-applied confirmation
            delta.confirm.append(item)

        # Clear the ones we applied
        delta.add = []
        delta.update = []

        # Keep conflicts and close items for validation
        needs_validation.conflict = delta.conflict
        needs_validation.close = delta.close

        # Deduplicate and save
        state = self._deduplicate_state(state)
        self._cs_repo.save(state)
        self._cs_repo.save_delta(
            patient_id, delta, document_id,
            auto_applied=auto_apply_low_risk,
        )

        return needs_validation

    def rebuild_from_events(self, patient_id: str) -> ClinicalState:
        """Rebuild all Clinical State views from events and evidence."""
        state = ClinicalState(patient_id=patient_id)
        events = self._event_repo.get_by_patient(patient_id)
        for event in sorted(events, key=lambda e: e.event_date):
            if event.status != "rejected":
                self._apply_event_to_state(state, event)

        # Evidence is the complete projection. Clinical Events remain a
        # specialized temporal view used for state transitions.
        self._apply_patient_evidence(state, patient_id)
        self._apply_patient_projections(state, patient_id)

        # Deduplicate and clean
        state = self._deduplicate_state(state)

        self._cs_repo.save(state)
        return state

    def _apply_patient_evidence(self, state: ClinicalState,
                                patient_id: str) -> None:
        if not self._evidence_repo:
            return
        evidence = self._evidence_repo.get_by_patient(patient_id)
        for item in evidence:
            self._apply_evidence_to_state(state, item)

    def _apply_patient_projections(self, state: ClinicalState,
                                   patient_id: str) -> None:
        if not self._projection_repo:
            return
        state.document_projections = [
            self._projection_state_view(projection)
            for projection in self._projection_repo.get_by_patient(patient_id)
        ]

    @staticmethod
    def _projection_state_view(projection) -> dict:
        """Avoid duplicating quotes/coordinates already held by observations."""
        return {
            "projection_id": projection.projection_id,
            "document_id": projection.document_id,
            "document_date": projection.document_date,
            "document_type": projection.document_type,
            "observations": [{
                key: value for key, value in observation.to_dict().items()
                if key not in {"source_spans", "data"}
            } for observation in projection.observations],
            "relationships": [
                relationship.to_dict()
                for relationship in projection.relationships
            ],
            "conflicts": [
                conflict.to_dict() for conflict in projection.conflicts
            ],
            "consolidation_method": projection.consolidation_method,
            "model_name": projection.model_name,
            "prompt_version": projection.prompt_version,
            "status": projection.status,
        }

    def _apply_evidence_to_state(self, state: ClinicalState, item) -> None:
        """Preserve every evidence item and update convenient state views."""
        observation = {
            "evidence_id": item.evidence_id,
            "date": item.observed_date,
            "category": item.category,
            "entity": item.normalized_entity,
            "assertion": item.assertion,
            "temporality": item.temporality,
            "clinical_status": item.clinical_status,
            "value_text": item.value_text,
            "numeric_value": item.numeric_value,
            "unit": item.unit,
            "source_document_id": item.document_id,
            "source_page": item.source_page,
            "source_text": item.source_text,
            "bbox": list(item.bbox) if item.bbox else None,
            "confidence": item.confidence,
            "extraction_method": item.extraction_method,
            "model_name": item.model_name,
            "status": item.status,
        }
        state.observations.append(observation)

        assertion = (item.assertion or "").lower()
        is_present = assertion not in {"absent", "negated", "not_present"}
        category = item.category

        if category == "medication_current" and is_present:
            if not any(
                treatment.name.lower() == item.normalized_entity.lower()
                for treatment in state.active_treatments
            ):
                state.active_treatments.append(Treatment(
                    name=item.normalized_entity,
                    start_date=item.observed_date,
                    source_document_id=item.document_id,
                    notes=item.source_text,
                ))
        elif category == "biomarker" and is_present:
            state.biomarkers.append(Biomarker(
                name=item.normalized_entity,
                value=item.value_text,
                date=item.observed_date,
                interpretation=item.clinical_status,
                source_event_id=item.evidence_id,
            ))
        elif category == "laboratory_finding":
            state.lab_trends.append({
                "date": item.observed_date,
                "parameter": item.normalized_entity,
                "value": item.numeric_value,
                "value_text": item.value_text,
                "unit": item.unit,
                "status": item.clinical_status,
                "reference": item.data.get("reference_text"),
                "evidence_id": item.evidence_id,
                "document_id": item.document_id,
                "page": item.source_page,
            })
        elif category == "care_plan" and is_present:
            state.follow_up.append({
                "date": item.observed_date,
                "description": item.normalized_entity,
                "evidence_id": item.evidence_id,
                "document_id": item.document_id,
            })
        elif (category == "symptom" and is_present and
              item.clinical_status not in {"resolved", "inactive"}):
            if item.normalized_entity not in state.open_problems:
                state.open_problems.append(item.normalized_entity)

    def _deduplicate_state(self, state: ClinicalState) -> ClinicalState:
        """Remove duplicate diagnoses and treatments, normalize names."""
        from difflib import SequenceMatcher

        # --- Deduplicate diagnoses ---
        seen = []
        unique_diagnoses = []
        for d in state.active_diagnoses:
            name_lower = d.name.lower().strip()
            # Skip garbage entries (too short, wrong dates)
            if len(name_lower) < 3:
                continue
            if d.date and len(d.date) < 4:  # "2" is not a valid date
                d.date = None
            # Check similarity with existing diagnoses
            is_dup = False
            for existing in unique_diagnoses:
                ratio = SequenceMatcher(None, name_lower, existing.name.lower()).ratio()
                if ratio > 0.75:
                    is_dup = True
                    # Keep the longer (more detailed) version
                    if len(name_lower) > len(existing.name):
                        existing.name = d.name
                    break
            if not is_dup:
                unique_diagnoses.append(d)

        # Move inactive ones to past
        state.active_diagnoses = []
        state.past_diagnoses = list(state.past_diagnoses or [])
        for d in unique_diagnoses:
            if d.status == "resolved":
                state.past_diagnoses.append(d)
            else:
                state.active_diagnoses.append(d)

        # --- Deduplicate treatments ---
        def _clean_treatment_name(name: str) -> str:
            """Normalize treatment names."""
            name = name.strip()
            # Remove generic prefixes
            for prefix in ['terapia ', 'trattamento ', 'farmaco ', 'agente ']:
                if name.lower().startswith(prefix):
                    name = name[len(prefix):]
            return name

        unique_active = []
        for t in state.active_treatments:
            name = _clean_treatment_name(t.name)
            if len(name) < 3:
                continue
            # Check for duplicates
            is_dup = False
            for existing in unique_active:
                if SequenceMatcher(None, name.lower(), existing.name.lower()).ratio() > 0.7:
                    is_dup = True
                    break
            if not is_dup:
                t.name = name
                unique_active.append(t)

        unique_completed = []
        for t in (state.completed_treatments or []):
            name = _clean_treatment_name(t.name)
            if len(name) < 3:
                continue
            is_dup = False
            for existing in unique_active + unique_completed:
                if SequenceMatcher(None, name.lower(), existing.name.lower()).ratio() > 0.7:
                    is_dup = True
                    break
            if not is_dup:
                t.name = name
                unique_completed.append(t)

        state.active_treatments = unique_active
        state.completed_treatments = unique_completed

        # --- Deduplicate toxicities ---
        unique_tox = []
        for t in state.toxicities:
            is_dup = False
            for existing in unique_tox:
                if SequenceMatcher(None, t.name.lower(), existing.name.lower()).ratio() > 0.7:
                    is_dup = True
                    if t.grade is not None and existing.grade is None:
                        existing.grade = t.grade
                    break
            if not is_dup and len(t.name) > 3:
                unique_tox.append(t)
        state.toxicities = unique_tox

        # --- Deduplicate procedures ---
        unique_proc = []
        for p in state.procedures:
            is_dup = False
            for existing in unique_proc:
                if SequenceMatcher(None, p.name.lower(), existing.name.lower()).ratio() > 0.75:
                    is_dup = True
                    break
            if not is_dup and len(p.name) > 3:
                unique_proc.append(p)
        state.procedures = unique_proc

        # Generic evidence views use stable IDs, so unlike fuzzy clinical
        # concepts they can be deduplicated exactly without loss.
        def _unique_by(items: list[dict], key_name: str) -> list[dict]:
            seen_ids = set()
            unique = []
            for item in items:
                item_id = item.get(key_name)
                if item_id and item_id in seen_ids:
                    continue
                if item_id:
                    seen_ids.add(item_id)
                unique.append(item)
            return unique

        state.observations = _unique_by(state.observations, "evidence_id")
        state.observations.sort(key=lambda item: (
            item.get("date") or "",
            item.get("source_document_id") or "",
            item.get("source_page") or 0,
        ))
        state.lab_trends = _unique_by(state.lab_trends, "evidence_id")
        state.follow_up = _unique_by(state.follow_up, "evidence_id")
        state.open_problems = list(dict.fromkeys(state.open_problems))

        return state

    def _apply_event_to_state(self, state: ClinicalState,
                              event: ClinicalEvent):
        """Apply a single event to the Clinical State."""
        etype = event.event_type

        if etype == "diagnosis":
            existing = [d for d in state.active_diagnoses
                       if d.name.lower() == event.entity.lower()]
            if not existing:
                state.active_diagnoses.append(Diagnosis(
                    name=event.entity,
                    date=event.event_date,
                    source_event_id=event.event_id,
                ))

        elif etype == "treatment_started":
            # Close conflicting treatments
            for t in state.active_treatments:
                if self._is_same_class(t.name, event.entity):
                    t.status = "completed"
                    t.end_date = event.event_date
                    state.completed_treatments.append(t)
            state.active_treatments = [t for t in state.active_treatments
                                       if t.status == "active"]
            state.active_treatments.append(Treatment(
                name=event.entity,
                start_date=event.event_date,
                source_event_id=event.event_id,
            ))

        elif etype == "treatment_ended":
            for t in state.active_treatments:
                if t.name.lower() == event.entity.lower():
                    t.status = "completed"
                    t.end_date = event.event_date
                    state.completed_treatments.append(t)
            state.active_treatments = [t for t in state.active_treatments
                                       if t.status == "active"]

        elif etype == "toxicity":
            state.toxicities.append(Toxicity(
                name=event.entity,
                grade=int(event.value) if event.value else None,
                date=event.event_date,
                source_event_id=event.event_id,
            ))

        elif etype == "procedure" or etype == "surgery":
            state.procedures.append(Procedure(
                name=event.entity,
                date=event.event_date,
                source_event_id=event.event_id,
            ))

        elif etype == "hospitalization":
            state.hospitalizations.append({
                "date": event.event_date,
                "description": event.entity,
                "event_id": event.event_id,
            })

        elif etype == "radiology_finding":
            state.imaging_findings.append({
                "date": event.event_date,
                "finding": event.entity,
                "event_id": event.event_id,
            })

    def _is_same_class(self, drug1: str, drug2: str) -> bool:
        """Crude check if two drugs might be in the same therapeutic class."""
        d1 = drug1.lower().split()[0]
        d2 = drug2.lower().split()[0]
        return d1 == d2

    def _deterministic_delta(self, current_state: ClinicalState,
                             new_events: list[ClinicalEvent]) -> ClinicalStateDelta:
        """Rule-based delta computation (no LLM)."""
        delta = ClinicalStateDelta()

        known_diagnoses = {d.name.lower() for d in current_state.active_diagnoses}
        known_treatments = {t.name.lower() for t in current_state.active_treatments}

        for event in new_events:
            if event.event_type == "diagnosis":
                if event.entity.lower() not in known_diagnoses:
                    delta.add.append({"type": "diagnosis", "entity": event.entity,
                                      "date": event.event_date})
                else:
                    delta.confirm.append({"type": "diagnosis", "entity": event.entity})

            elif event.event_type == "treatment_started":
                if event.entity.lower() not in known_treatments:
                    delta.add.append({"type": "treatment", "entity": event.entity,
                                      "date": event.event_date})
                else:
                    delta.confirm.append({"type": "treatment", "entity": event.entity})

            elif event.event_type == "toxicity":
                delta.add.append({"type": "toxicity", "entity": event.entity,
                                  "grade": event.value, "date": event.event_date})

            elif event.event_type in ("procedure", "surgery"):
                delta.add.append({"type": "procedure", "entity": event.entity,
                                  "date": event.event_date})

        return delta

    def _llm_delta(self, current_state: ClinicalState,
                   new_events: list[ClinicalEvent]) -> ClinicalStateDelta:
        """LLM-based delta computation via the Clinical State model."""
        if not self._llm_client:
            return self._deterministic_delta(current_state, new_events)

        state_dict = current_state.to_dict()
        events_dict = [e.to_dict() for e in new_events]

        delta_dict = self._llm_client.propose_clinical_state_delta(
            state_dict, events_dict
        )

        return ClinicalStateDelta.from_dict(delta_dict)

    def _apply_delta_item(self, state: ClinicalState, action: str, item: dict):
        """Apply a single delta item to the state."""
        itype = item.get("type", "")

        if action == "add":
            if itype == "diagnosis":
                state.active_diagnoses.append(Diagnosis(
                    name=item["entity"], date=item.get("date"),
                ))
            elif itype == "treatment":
                state.active_treatments.append(Treatment(
                    name=item["entity"], start_date=item.get("date"),
                ))
            elif itype == "toxicity":
                state.toxicities.append(Toxicity(
                    name=item["entity"], grade=item.get("grade"),
                    date=item.get("date"),
                ))
            elif itype == "procedure":
                state.procedures.append(Procedure(
                    name=item["entity"], date=item.get("date"),
                ))

        elif action == "close":
            if itype == "diagnosis":
                for d in state.active_diagnoses:
                    if d.name.lower() == item.get("entity", "").lower():
                        state.past_diagnoses.append(d)
                state.active_diagnoses = [d for d in state.active_diagnoses
                                          if d.name.lower() != item.get("entity", "").lower()]
            elif itype == "treatment":
                for t in state.active_treatments:
                    if t.name.lower() == item.get("entity", "").lower():
                        t.status = "completed"
                        t.end_date = item.get("date")
                        state.completed_treatments.append(t)
                state.active_treatments = [t for t in state.active_treatments
                                           if t.status == "active"]

        elif action == "update":
            if itype == "staging":
                state.staging = item.get("value")
